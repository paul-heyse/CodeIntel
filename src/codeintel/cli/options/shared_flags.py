"""Shared flag helpers that preserve per-command env var names."""

from __future__ import annotations

from dataclasses import field, make_dataclass
from pathlib import Path
from typing import Annotated, Protocol, cast

from cyclopts import Parameter

from codeintel.cli.options.registry import JSON_FLAG, OUTPUT_FORMAT, PROJECT_ROOT, VERBOSE
from codeintel.cli.options.types import CommandPath, option_param
from codeintel.cli.rendering.types import OutputFormat
from codeintel.observability.cli import RunContext


class SharedFlagsProtocol(Protocol):
    """Protocol for shared CLI flags injected into commands."""

    project_root: Path | None
    output_format: OutputFormat
    json: bool
    verbose: int
    run_context: RunContext | None


_SHARED_FLAGS_CACHE: dict[CommandPath, type[SharedFlagsProtocol]] = {}


def _shared_flags_metadata() -> dict[str, Parameter]:
    return {"parameter": Parameter(name="*")}


def _shared_flags_class_name(command_path: CommandPath) -> str:
    safe_parts = (part.replace("-", "_") for part in command_path)
    return "SharedFlags_" + "_".join(safe_parts)


def shared_flags_type(command_path: CommandPath) -> type[SharedFlagsProtocol]:
    """Return a cached SharedFlags dataclass for a command path.

    Returns
    -------
    type[SharedFlagsProtocol]
        Generated SharedFlags dataclass type for the command path.
    """
    cached = _SHARED_FLAGS_CACHE.get(command_path)
    if cached is not None:
        return cached

    fields: list[tuple[str, object, object]] = [
        (
            "project_root",
            Annotated[Path | None, option_param(PROJECT_ROOT, command_path=command_path)],
            field(default=None),
        ),
        (
            "output_format",
            Annotated[OutputFormat, option_param(OUTPUT_FORMAT, command_path=command_path)],
            field(default=OutputFormat.TEXT),
        ),
        (
            "json",
            Annotated[bool, option_param(JSON_FLAG, command_path=command_path)],
            field(default=False),
        ),
        (
            "verbose",
            Annotated[int, option_param(VERBOSE, command_path=command_path)],
            field(default=0),
        ),
        (
            "run_context",
            Annotated[RunContext | None, Parameter(parse=False)],
            field(default=None),
        ),
    ]

    cls = make_dataclass(
        _shared_flags_class_name(command_path),
        fields,
        frozen=True,
        slots=True,
    )
    _SHARED_FLAGS_CACHE[command_path] = cast("type[SharedFlagsProtocol]", cls)
    return cast("type[SharedFlagsProtocol]", cls)


def shared_flags_field(command_path: CommandPath) -> SharedFlagsProtocol:
    """Create a SharedFlags field with Cyclopts metadata for nested flattening.

    Returns
    -------
    SharedFlagsProtocol
        Field instance configured for nested parameter flattening.
    """
    flags_type = shared_flags_type(command_path)
    return field(default_factory=flags_type, metadata=_shared_flags_metadata())


__all__ = [
    "SharedFlagsProtocol",
    "shared_flags_field",
    "shared_flags_type",
]
