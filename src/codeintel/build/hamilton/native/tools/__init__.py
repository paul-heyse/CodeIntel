"""Tool execution abstraction for native Hamilton targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.hamilton.io.artifact_ref import ArtifactRef


@dataclass(frozen=True)
class ToolExecutionSpec:
    """Specification for external tool execution.

    Attributes
    ----------
    tool_name
        Name of the tool executable to run.
    command_args
        Command-line arguments to pass to the tool.
    output_path
        Expected output file path from tool execution.
    timeout_seconds
        Maximum execution time in seconds.
    env_vars
        Optional environment variables to set for tool execution.
    """

    tool_name: str
    command_args: tuple[str, ...]
    output_path: Path
    timeout_seconds: float = 300.0
    env_vars: dict[str, str] | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    """Result of external tool execution.

    Attributes
    ----------
    success
        Whether tool execution succeeded (return code 0).
    artifact
        Artifact reference if execution succeeded and output exists.
    duration_ms
        Execution duration in milliseconds.
    stdout
        Standard output from tool execution.
    stderr
        Standard error from tool execution.
    return_code
        Process return code (0 for success).
    """

    success: bool
    artifact: ArtifactRef | None
    duration_ms: float
    stdout: str
    stderr: str
    return_code: int


__all__ = [
    "ToolExecutionResult",
    "ToolExecutionSpec",
]
