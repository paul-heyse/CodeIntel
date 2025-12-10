"""Build command operation specifications.

Define operation specs for the build command group including
status, run, and history commands.

Note: These register to the LEGACY registry for backward compatibility.
New handler registrations are in handlers/build.py (NEW registry).
"""

from __future__ import annotations

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import BuildHistoryResult, BuildStatusResult
from codeintel.cli.execution import OperationCategory, OperationSpec
from codeintel.cli.introspection.registry import register_operation


def _build_status_handler() -> CliResult[BuildStatusResult]:
    """Show build target status handler.

    Returns
    -------
    CliResult[BuildStatusResult]
        Build status result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    return CliResult.ok(
        BuildStatusResult(
            targets=[],
            stale_count=0,
            fresh_count=0,
        )
    )


def _build_history_handler(
    *,
    run_id: str | None = None,
    limit: int = 10,
) -> CliResult[BuildHistoryResult]:
    """Show build execution history handler.

    Parameters
    ----------
    run_id
        Specific run ID to show details for.
    limit
        Maximum number of history entries to return.

    Returns
    -------
    CliResult[BuildHistoryResult]
        Build history result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    The limit and run_id parameters will be used by the actual impl.
    """
    _ = run_id  # Will filter to specific run in actual impl
    return CliResult.ok(
        BuildHistoryResult(
            runs=[],
            count=min(0, limit),
        )
    )


# Build Status Operation (registers to LEGACY registry)
BUILD_STATUS_SPEC: OperationSpec[BuildStatusResult] = register_operation(
    OperationSpec(
        operation_id="build.status",
        handler=_build_status_handler,
        category=OperationCategory.BUILD,
        param_schema=None,
        requires_progress=False,
        description="Show build target status",
    )
)

# Build History Operation (registers to LEGACY registry)
BUILD_HISTORY_SPEC: OperationSpec[BuildHistoryResult] = register_operation(
    OperationSpec(
        operation_id="build.history",
        handler=_build_history_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="Show build execution history",
    )
)

__all__ = [
    "BUILD_HISTORY_SPEC",
    "BUILD_STATUS_SPEC",
]
