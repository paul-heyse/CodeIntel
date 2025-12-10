"""IDE operation specifications.

Define and register operations for the ide command group including
status and config commands.
"""

from __future__ import annotations

from codeintel.cli.execution import OperationCategory, OperationSpec
from codeintel.cli.introspection import register_operation
from codeintel.cli.core.result_types import IdeConfigResult, IdeStatusResult
from codeintel.cli.core import CliResult


def _ide_status_handler() -> CliResult[IdeStatusResult]:
    """Check IDE connection status handler.

    Returns
    -------
    CliResult[IdeStatusResult]
        IDE status result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    return CliResult.ok(
        IdeStatusResult(
            connected=False,
            ide_type=None,
            workspace_path=None,
            extensions=[],
        )
    )


def _ide_config_handler(
    *,
    key: str | None = None,
) -> CliResult[IdeConfigResult]:
    """Get IDE configuration handler.

    Parameters
    ----------
    key
        Specific config key to retrieve.

    Returns
    -------
    CliResult[IdeConfigResult]
        IDE config result.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    # Use parameter to avoid unused variable warning
    _ = key
    return CliResult.ok(
        IdeConfigResult(
            settings={},
            path=None,
        )
    )


# IDE Status Operation
IDE_STATUS_SPEC: OperationSpec[IdeStatusResult] = register_operation(
    OperationSpec(
        operation_id="ide.status",
        handler=_ide_status_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="Check IDE connection status",
    )
)

# IDE Config Operation
IDE_CONFIG_SPEC: OperationSpec[IdeConfigResult] = register_operation(
    OperationSpec(
        operation_id="ide.config",
        handler=_ide_config_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="Show IDE configuration",
    )
)

__all__ = [
    "IDE_CONFIG_SPEC",
    "IDE_STATUS_SPEC",
]
