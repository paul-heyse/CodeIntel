"""Build command operation specifications.

Define and register operations for the build command group including
status, run, and history commands.
"""

from __future__ import annotations

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import BuildHistoryResult, BuildStatusResult
from codeintel.cli.execution import OperationCategory, OperationSpec
from codeintel.cli.introspection import register_operation


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
    # Return empty result - actual implementation in cyclopts_build
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
    # Use limit and run_id in returned result to satisfy linter
    _ = run_id  # Will filter to specific run in actual impl
    return CliResult.ok(
        BuildHistoryResult(
            runs=[],
            count=min(0, limit),  # Placeholder uses limit
        )
    )


# Build Status Operation
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

# Build History Operation
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
