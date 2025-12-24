"""Helpers for auditing real tool invocations in tests."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import ToolsConfig
from codeintel.core.tools import ToolName

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Single tool invocation record captured from ToolRunner."""

    tool: str
    argv: tuple[str, ...]
    cwd: str | None
    env_keys: tuple[str, ...]
    started_at: datetime | None
    duration_s: float
    returncode: int
    version: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallLog:
    """Wrapper around a JSONL tool call log file."""

    path: Path

    def read(self) -> list[ToolCall]:
        """Load tool call records from the JSONL file.

        Returns
        -------
        list[ToolCall]
            Parsed tool call records.
        """
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [_parse_tool_call(line) for line in lines]


def require_tooling(tools_config: ToolsConfig | None = None) -> dict[ToolName, Path]:
    """Verify configured tool binaries exist and are executable.

    Parameters
    ----------
    tools_config
        Optional explicit ToolsConfig; defaults to ToolsConfig.default().

    Returns
    -------
    dict[ToolName, Path]
        Mapping of tool names to resolved executable paths.

    Raises
    ------
    RuntimeError
        If any required tool binary is missing or not executable.
    """
    config = tools_config or ToolsConfig.default()
    resolved: dict[ToolName, Path] = {}
    missing: list[str] = []
    for tool in ToolName:
        configured = config.resolve_path(tool)
        candidate = Path(configured)
        if candidate.is_file():
            resolved_path = candidate
        else:
            discovered = shutil.which(configured)
            resolved_path = Path(discovered) if discovered else None
        if resolved_path is None or not os.access(resolved_path, os.X_OK):
            missing.append(f"{tool.value}={configured!r}")
            continue
        resolved[tool] = resolved_path
    if missing:
        message = "Missing tool binaries: " + ", ".join(missing)
        raise RuntimeError(message)
    return resolved


def assert_tool_called(
    calls: Iterable[ToolCall],
    tool: ToolName | str,
    *,
    expected_args_contains: Sequence[str] | None = None,
    fail_if_missing: Sequence[str] | None = None,
    min_calls: int = 1,
) -> ToolCall:
    """Assert a tool was called with expected arguments.

    Parameters
    ----------
    calls
        Iterable of ToolCall records.
    tool
        Tool name to match.
    expected_args_contains
        Arguments that must appear in the invocation.
    fail_if_missing
        Required arguments; failure message lists missing ones.
    min_calls
        Minimum number of calls required for the tool.

    Returns
    -------
    ToolCall
        The first matching call.

    Raises
    ------
    AssertionError
        If the tool was not called or required arguments were missing.
    """
    tool_name = tool.value if isinstance(tool, ToolName) else str(tool)
    matches = [call for call in calls if call.tool == tool_name]
    if len(matches) < min_calls:
        message = f"Expected {tool_name} to be called at least {min_calls} times"
        raise AssertionError(message)
    selected = matches[0]
    argv = selected.argv
    expected = list(expected_args_contains or [])
    missing = [arg for arg in expected if arg not in argv]
    if missing:
        message = f"Missing expected args for {tool_name}: {missing}"
        raise AssertionError(message)
    required = list(fail_if_missing or [])
    required_missing = [arg for arg in required if arg not in argv]
    if required_missing:
        message = f"Missing required args for {tool_name}: {required_missing}"
        raise AssertionError(message)
    return selected


def _parse_tool_call(line: str) -> ToolCall:
    raw = json.loads(line)
    started_at = _parse_started_at(raw.get("started_at"))
    return ToolCall(
        tool=str(raw.get("tool", "")),
        argv=tuple(raw.get("argv") or []),
        cwd=raw.get("cwd"),
        env_keys=tuple(raw.get("env_keys") or []),
        started_at=started_at,
        duration_s=float(raw.get("duration_s", 0.0)),
        returncode=int(raw.get("returncode", 0)),
        version=raw.get("version"),
    )


def _parse_started_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


__all__ = [
    "ToolCall",
    "ToolCallLog",
    "assert_tool_called",
    "require_tooling",
]
