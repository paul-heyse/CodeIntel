"""Structured runners for external tools with caching and typed results."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ToolName = Literal["pyright", "pyrefly", "coverage", "git", "ruff", "pytest"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    """Structured output from a tool invocation."""

    tool: ToolName
    returncode: int
    stdout: str
    stderr: str
    output_path: Path | None = None
    duration_s: float | None = None

    @property
    def ok(self) -> bool:
        """Return True when the tool completed successfully."""
        return self.returncode == 0


class ToolRunner:
    """Run external tools with optional caching."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        tool: ToolName,
        args: list[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
    ) -> ToolResult:
        """
        Execute a tool process and capture stdout/stderr.

        Parameters
        ----------
        tool:
            Name of the tool being invoked (validated against whitelist).
        args:
            Argument vector including the executable.
        cwd:
            Optional working directory.
        output_path:
            Optional path to persist output; when provided the tool is expected
            to write to that location.

        Returns
        -------
        ToolResult
            Structured result with stdout/stderr and return code.

        Raises
        ------
        ValueError
            If args is empty or executable is not in the allowed tool set.
        """
        if not args:
            message = "args must include the executable path/name"
            raise ValueError(message)
        if tool not in {"pyright", "pyrefly", "coverage", "git", "ruff", "pytest"}:
            message = f"Unknown tool {tool}"
            raise ValueError(message)
        exe = args[0]
        if shutil.which(exe) is None:
            stdout = ""
            stderr = f"{exe} not found"
            return ToolResult(
                tool=tool, returncode=127, stdout=stdout, stderr=stderr, duration_s=0.0
            )
        if output_path is None:
            output_path = self.cache_dir / f"{tool}.out"

        async def _exec() -> ToolResult:
            start = asyncio.get_event_loop().time()
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(cwd) if cwd is not None else None,
                stdout=PIPE,
                stderr=PIPE,
            )
            stdout_b, stderr_b = await proc.communicate()
            duration = asyncio.get_event_loop().time() - start
            return ToolResult(
                tool=tool,
                returncode=proc.returncode if proc.returncode is not None else 1,
                stdout=stdout_b.decode(),
                stderr=stderr_b.decode(),
                output_path=output_path,
                duration_s=duration,
            )

        return asyncio.run(_exec())

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
