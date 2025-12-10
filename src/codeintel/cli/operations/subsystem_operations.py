"""Subsystem operation specifications.

Define operation specs for the subsystem command group including
list and show commands.

Note: These register to the LEGACY registry for backward compatibility.
New handler registrations are in handlers/subsystem.py (NEW registry).
"""

from __future__ import annotations

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import SubsystemDetailResult, SubsystemListResult
from codeintel.cli.execution import OperationCategory, OperationSpec
from codeintel.cli.introspection.registry import register_operation


def _subsystem_list_handler(
    *,
    include_deps: bool = False,
) -> CliResult[SubsystemListResult]:
    """List subsystems handler.

    Parameters
    ----------
    include_deps
        Include dependency information.

    Returns
    -------
    CliResult[SubsystemListResult]
        Subsystem list result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    _ = include_deps
    return CliResult.ok(
        SubsystemListResult(
            subsystems=[],
            count=0,
        )
    )


def _subsystem_show_handler(
    *,
    name: str,
) -> CliResult[SubsystemDetailResult]:
    """Show subsystem details handler.

    Parameters
    ----------
    name
        Subsystem name.

    Returns
    -------
    CliResult[SubsystemDetailResult]
        Subsystem detail result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    return CliResult.ok(
        SubsystemDetailResult(
            name=name,
            description=None,
            modules=[],
            dependencies=[],
            metrics={},
        )
    )


# Subsystem List Operation (registers to LEGACY registry)
SUBSYSTEM_LIST_SPEC: OperationSpec[SubsystemListResult] = register_operation(
    OperationSpec(
        operation_id="subsystem.list",
        handler=_subsystem_list_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="List subsystems",
    )
)

# Subsystem Show Operation (registers to LEGACY registry)
SUBSYSTEM_SHOW_SPEC: OperationSpec[SubsystemDetailResult] = register_operation(
    OperationSpec(
        operation_id="subsystem.show",
        handler=_subsystem_show_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="Show subsystem details",
    )
)

__all__ = [
    "SUBSYSTEM_LIST_SPEC",
    "SUBSYSTEM_SHOW_SPEC",
]
