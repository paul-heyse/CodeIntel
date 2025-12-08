"""Asynchronous command execution helpers for mkdocs tooling."""

from __future__ import annotations

import asyncio
from asyncio.subprocess import PIPE, STDOUT
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CommandError(RuntimeError):
    """Raised when an external command fails."""

    cmd: tuple[str, ...]
    returncode: int
    output: str

    def __init__(
        self,
        cmd: Sequence[str],
        returncode: int,
        output: str | None = None,
    ) -> None:
        message = f"Command failed ({returncode}): {' '.join(cmd)}"
        detail = f"{message}\n{output}" if output else message
        super().__init__(detail)
        self.cmd = tuple(cmd)
        self.returncode = returncode
        self.output = output or ""


def _copy_env(env: Mapping[str, str] | None) -> dict[str, str] | None:
    """Return a shallow copy of the environment mapping.

    Returns
    -------
    dict[str, str] | None
        Copied environment mapping or None when no environment provided.
    """
    if env is None:
        return None
    return dict(env)


async def _run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path | None,
    env: Mapping[str, str] | None,
    merge_stderr: bool,
) -> tuple[int, bytes | None, bytes | None]:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        env=_copy_env(env),
        stdout=PIPE,
        stderr=STDOUT if merge_stderr else PIPE,
    )
    stdout, stderr = await process.communicate()
    returncode = process.returncode
    if returncode is None:
        raise CommandError(cmd, -1, "Process exited without return code")
    return returncode, stdout, stderr


async def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    merge_stderr: bool = True,
) -> str:
    """Run a command and return combined stdout (and optionally stderr).

    Parameters
    ----------
    cmd
        Command and arguments to execute.
    cwd
        Working directory for the command.
    env
        Environment variables for the command.
    merge_stderr
        If True, merge stderr into stdout.

    Returns
    -------
    str
        Combined command output.

    Raises
    ------
    CommandError
        If the command exits with a non-zero status.
    """
    returncode, stdout, stderr = await _run_subprocess(
        cmd,
        cwd=cwd,
        env=env,
        merge_stderr=merge_stderr,
    )
    output = (stdout or b"").decode() + ((stderr or b"").decode() if not merge_stderr else "")
    if returncode != 0:
        raise CommandError(cmd, returncode, output)
    return output


async def stream_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    merge_stderr: bool = True,
) -> list[str]:
    """Run a command and stream output line by line.

    Parameters
    ----------
    cmd
        Command and arguments to execute.
    cwd
        Working directory for the command.
    env
        Environment variables for the command.
    merge_stderr
        If True, merge stderr into stdout.

    Returns
    -------
    list[str]
        Ordered output lines from the command.

    Raises
    ------
    CommandError
        If the command exits with a non-zero status.
    """
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        env=_copy_env(env),
        stdout=PIPE,
        stderr=STDOUT if merge_stderr else PIPE,
    )
    lines: list[str] = []
    if process.stdout is not None:
        async for raw in process.stdout:
            line = raw.decode().rstrip()
            if line:
                lines.append(line)
    if process.stderr is not None and not merge_stderr:
        stderr_output = (await process.stderr.read()).decode()
        if stderr_output:
            lines.extend(stderr_output.splitlines())
    await process.wait()
    output = "\n".join(lines)
    returncode = process.returncode
    if returncode is None:
        raise CommandError(cmd, -1, output)
    if returncode != 0:
        raise CommandError(cmd, returncode, output)
    return lines


def run_command_sync(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    merge_stderr: bool = True,
) -> str:
    """Run a command synchronously.

    Returns
    -------
    str
        Combined command output.

    Raises
    ------
    CommandError
        If the command exits with a non-zero status.
    """
    try:
        return asyncio.run(run_command(cmd, cwd=cwd, env=env, merge_stderr=merge_stderr))
    except CommandError as exc:
        raise CommandError(exc.cmd, exc.returncode, exc.output) from exc


def stream_command_sync(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    merge_stderr: bool = True,
) -> list[str]:
    """Stream command output synchronously.

    Returns
    -------
    list[str]
        Ordered output lines from the command.

    Raises
    ------
    CommandError
        If the command exits with a non-zero status.
    """
    try:
        return asyncio.run(stream_command(cmd, cwd=cwd, env=env, merge_stderr=merge_stderr))
    except CommandError as exc:
        raise CommandError(exc.cmd, exc.returncode, exc.output) from exc
