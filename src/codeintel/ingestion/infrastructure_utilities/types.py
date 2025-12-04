"""Canonical type definitions for ingestion infrastructure.

This module provides the single source of truth for foundational types used
across the ingestion system. All other modules should import from here to
avoid duplication and ensure consistency.

Types
-----
ToolStatus
    Normalized status for external tool invocations. Used by tool plugins,
    adapters, and port result types.
"""

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

    Examples
    --------
    >>> from codeintel.ingestion.infrastructure_utilities.types import ToolStatus
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


__all__ = ["ToolStatus"]
