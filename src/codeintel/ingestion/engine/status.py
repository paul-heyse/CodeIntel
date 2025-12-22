"""Tool execution status values."""

from __future__ import annotations

from enum import StrEnum


class ToolStatus(StrEnum):
    """Normalized status for external tool invocations.

    This enum represents the possible outcomes of running an external tool
    (pyright, ruff, coverage, scip-python, pytest, etc.) via the tool plugin
    system.

    Members
    -------
    OK
        Tool executed successfully and produced valid output.
    NOT_FOUND
        Tool binary was not found on the system PATH.
    FAILED
        Tool execution failed (non-zero exit, parse error, or exception).
    TIMEOUT
        Tool execution exceeded the configured timeout.
    SKIPPED
        Tool execution was skipped (tool not available or not applicable).

    Examples
    --------
    >>> from codeintel.ingestion.engine.status import ToolStatus
    >>> status = ToolStatus.OK
    >>> status == "ok"
    True
    >>> status.value
    'ok'
    """

    OK = "ok"
    NOT_FOUND = "not_found"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


__all__ = ["ToolStatus"]
