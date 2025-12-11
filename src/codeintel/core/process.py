"""Safe command execution helpers with allowlists and structured results.

This module centralizes external command execution to avoid scattered
subprocess usage, enforce allowlisted binaries, and capture structured
results for telemetry and diagnostics. It uses asyncio-based process
creation to avoid the security pitfalls flagged by Ruff's subprocess
rules while still supporting synchronous call sites via small wrappers.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


class CommandNotAllowedError(ValueError):
    """Raised when attempting to execute a command outside the allowlist."""

    def __init__(self, command: str) -> None:
        super().__init__(f"Command not allowed: {command}")


class CommandExecutionError(RuntimeError):
    """Raised when an allowlisted command exits with a non-zero status."""

    def __init__(self, cmd: Sequence[str], returncode: int, stderr: str) -> None:
        joined = " ".join(cmd)
        super().__init__(f"Command failed ({returncode}): {joined}\n{stderr}")
        self.command = tuple(cmd)
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class CommandResult:
    """Structured result for an executed command."""

    command: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int
    duration_seconds: float
    cwd: Path | None


class CommandExecutor:
    """Execute allowlisted commands with structured results and logging."""

    def __init__(
        self,
        *,
        allowed_commands: dict[str, Path],
        logger: logging.Logger | None = None,
    ) -> None:
        self._allowed_commands = allowed_commands
        self._logger = logger or logging.getLogger(__name__)

    @classmethod
    def for_build_tools(cls) -> CommandExecutor:
        """Create executor configured for build-time tooling.

        Returns
        -------
        CommandExecutor
            Executor with scip and git allowlisted.
        """
        allowed = cls._resolve_allowlist(["scip-python", "scip", "git"])
        return cls(allowed_commands=allowed)

    def run_scip_index(
        self,
        repo_root: Path,
        output_path: Path,
        *,
        project_name: str = "codeintel",
    ) -> CommandResult:
        """Run scip-python index with strict allowlisting.

        Returns
        -------
        CommandResult
            Structured execution result.
        """
        command = (
            "scip-python",
            "index",
            str(repo_root),
            "--project-name",
            project_name,
            "--output",
            str(output_path),
        )
        return self._run_checked(command, cwd=repo_root)

    def export_scip_to_json(self, scip_index: Path, json_path: Path) -> CommandResult:
        """Render SCIP index to JSON.

        Returns
        -------
        CommandResult
            Structured execution result.
        """
        command = ("scip", "print", str(scip_index), "--json")
        return self._run_checked(command, cwd=scip_index.parent, stdout_path=json_path)

    def read_git_revision(self, repo_root: Path) -> str:
        """Read the current git HEAD short SHA.

        Returns
        -------
        str
            Raw git revision string.
        """
        command = ("git", "rev-parse", "HEAD")
        result = self._run_checked(command, cwd=repo_root)
        return result.stdout.strip()

    def _run_checked(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        stdout_path: Path | None = None,
    ) -> CommandResult:
        """Execute an allowlisted command and raise on failure.

        Returns
        -------
        CommandResult
            Structured execution result.

        Raises
        ------
        CommandExecutionError
            If the command exits with a non-zero status code.
        """
        result = self.run(command, cwd=cwd, stdout_path=stdout_path)
        if result.returncode != 0:
            raise CommandExecutionError(result.command, result.returncode, result.stderr)
        return result

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        stdout_path: Path | None = None,
    ) -> CommandResult:
        """Execute an allowlisted command.

        Returns
        -------
        CommandResult
            Structured execution result.
        """
        prepared = self._prepare_command(command)
        start_time = time.perf_counter()
        stdout, stderr, returncode = asyncio.run(
            self._exec(prepared, cwd=cwd, stdout_path=stdout_path)
        )
        duration_seconds = time.perf_counter() - start_time
        result = CommandResult(
            command=prepared,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            duration_seconds=duration_seconds,
            cwd=cwd,
        )
        self._logger.debug(
            "Ran command %s (rc=%s, duration=%.3fs)",
            " ".join(prepared),
            returncode,
            duration_seconds,
        )
        return result

    @staticmethod
    def _resolve_allowlist(commands: Iterable[str]) -> dict[str, Path]:
        resolved: dict[str, Path] = {}
        for cmd in commands:
            path = shutil.which(cmd)
            if path is None:
                continue
            resolved[cmd] = Path(path)
        return resolved

    def _prepare_command(self, command: Sequence[str]) -> tuple[str, ...]:
        if not command:
            msg = "Empty command is not allowed"
            raise ValueError(msg)
        binary = command[0]
        allowed_path = self._allowed_commands.get(binary)
        if allowed_path is None:
            raise CommandNotAllowedError(binary)
        args = [str(part) for part in command[1:]]
        return (str(allowed_path), *args)

    @staticmethod
    async def _exec(
        command: Sequence[str],
        *,
        cwd: Path | None,
        stdout_path: Path | None,
    ) -> tuple[str, str, int]:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout_text = stdout_bytes.decode(errors="replace")
        stderr_text = stderr_bytes.decode(errors="replace")

        if stdout_path is not None:
            await asyncio.to_thread(stdout_path.write_text, stdout_text, encoding="utf-8")

        return_code = process.returncode
        if return_code is None:
            message = "Process exited without a return code"
            raise RuntimeError(message)

        return stdout_text, stderr_text, return_code


__all__ = [
    "CommandExecutionError",
    "CommandExecutor",
    "CommandNotAllowedError",
    "CommandResult",
]
