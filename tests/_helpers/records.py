"""Reusable call record types and recorder utility for fakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class ToolRunCall:
    """Record of a tool runner invocation."""

    tool: str
    args: list[str]
    cwd: Path
    timeout_ms: int | None
    env: dict[str, str] | None


@dataclass(frozen=True)
class ScipIndexCall:
    """Record of a SCIP indexer invocation."""

    repo_root: Path
    output_path: Path
    include_patterns: Sequence[str] | None
    exclude_patterns: Sequence[str] | None


@dataclass(frozen=True)
class ScipParseCall:
    """Record of a SCIP parse invocation."""

    scip_path: Path
    output_json_path: Path


@dataclass(frozen=True)
class TypeCheckCall:
    """Record of a type checker invocation."""

    repo_root: Path
    paths: Sequence[Path] | None
    config_path: Path | None


@dataclass(frozen=True)
class GitLogCall:
    """Record of a git log invocation."""

    repo_root: Path
    path: Path | None
    max_count: int | None
    since: str | None
    until: str | None


@dataclass(frozen=True)
class GitBlameCall:
    """Record of a git blame invocation."""

    repo_root: Path
    path: Path
    start_line: int | None
    end_line: int | None


@dataclass(frozen=True)
class CollectCall:
    """Record of a single-path collection invocation."""

    path: Path


@dataclass(frozen=True)
class StorageOpCall:
    """Record of a storage operation."""

    op: str
    target: str
    details: object


CallT = TypeVar("CallT")


class CallRecorder[CallT]:
    """Utility to capture and inspect call records in fakes."""

    calls: list[CallT]

    def __init__(self) -> None:
        self.calls = []

    def record(self, call: CallT) -> None:
        """Append a call record."""
        self.calls.append(call)

    @property
    def last_call(self) -> CallT | None:
        """Return the most recent call or None."""
        if not self.calls:
            return None
        return self.calls[-1]

    def clear(self) -> None:
        """Clear recorded calls."""
        self.calls.clear()
