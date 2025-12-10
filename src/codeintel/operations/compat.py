"""TEMPORARY compatibility shims for handler-to-operation migration.

!!! WARNING !!!
This module provides temporary bridges between the old HandlerContext-based
handlers and the new Operation-based system. ALL code in this module will be
DELETED once migration is complete.

Do NOT add new code that depends on these shims. Use Operations directly.
!!! WARNING !!!
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeGuard

from codeintel.cli.config import load_config
from codeintel.cli.core.results import CliResult
from codeintel.cli.errors import ProblemDetail as CliProblemDetail
from codeintel.cli.handlers.context import HandlerContext
from codeintel.operations.base import Capability, Operation, OperationSpec
from codeintel.operations.context import OpContext
from codeintel.operations.errors.problem_detail import ProblemDetail
from codeintel.operations.registry import get_default_registry
from codeintel.operations.result import Result

if TYPE_CHECKING:
    from collections.abc import Callable


LOG = logging.getLogger(__name__)


class _DataclassInstance(Protocol):
    __dataclass_fields__: dict[str, object]


def _is_dataclass_instance(value: object) -> TypeGuard[_DataclassInstance]:
    """Return True when value is a dataclass instance."""
    return is_dataclass(value) and not isinstance(value, type)


@dataclass(frozen=True)
class HandlerWrapperConfig:
    """Configuration for wrapping a handler as an operation."""

    operation_id: str
    params_type: type[object]
    result_type: type[object]
    capabilities: frozenset[str] = frozenset()
    description: str | None = None


def operation_from_handler(
    config: HandlerWrapperConfig,
    handler: Callable[[HandlerContext], CliResult[object]],
) -> type[Operation[object, object]]:
    """Wrap a legacy handler as an Operation class.

    TEMPORARY: This function will be removed after migration.

    Parameters
    ----------
    config
        Wrapper configuration with operation_id, types, and capabilities.
    handler
        Legacy handler function that takes HandlerContext.

    Returns
    -------
    type[Operation[object, object]]
        An Operation class wrapping the handler.
    """

    @dataclass
    class WrappedOperation(Operation[object, object]):
        """Wrapped handler operation."""

        __operation_id__: ClassVar[str] = config.operation_id
        __params_type__: ClassVar[type[object]] = config.params_type
        __result_type__: ClassVar[type[object]] = config.result_type
        __capabilities__: ClassVar[frozenset[str]] = config.capabilities

        def execute(self, params: object, ctx: OpContext) -> Result[object]:
            """Execute wrapped handler.

            Parameters
            ----------
            params
                Operation parameters.
            ctx
                Operation context.

            Returns
            -------
            Result[object]
                Operation result.
            """
            _ = (self, ctx)  # Instance method for protocol compatibility
            # Convert params to HandlerContext
            handler_ctx = _params_to_handler_context(params)

            # Execute the handler
            cli_result = handler(handler_ctx)

            # Convert CliResult to Result
            return result_from_cli_result(cli_result)

    # Set metadata
    WrappedOperation.__doc__ = config.description or f"Wrapped handler: {config.operation_id}"
    WrappedOperation.__name__ = f"Wrapped_{handler.__name__}"

    # Register with the registry
    group = config.operation_id.split(".", maxsplit=1)[0]
    spec = OperationSpec(
        operation_id=config.operation_id,
        name=WrappedOperation.__name__,
        description=config.description or f"Wrapped: {config.operation_id}",
        params_type=config.params_type,
        result_type=config.result_type,
        operation_class=WrappedOperation,
        group=group,
        capabilities=config.capabilities,
        require_storage=(
            Capability.STORAGE_READ in config.capabilities
            or Capability.STORAGE_WRITE in config.capabilities
        ),
    )

    get_default_registry().register(spec)

    return WrappedOperation


def _params_to_handler_context(params: object) -> HandlerContext:
    """Convert operation params to a HandlerContext.

    Parameters
    ----------
    params
        Operation parameters dataclass.

    Returns
    -------
    HandlerContext
        Handler context for legacy code.
    """
    from codeintel.cli.config import load_config
    from codeintel.cli.handlers.context import HandlerContext

    params_dict: dict[str, object] = {}
    if _is_dataclass_instance(params):
        for fld in fields(params):
            params_dict[fld.name] = getattr(params, fld.name)

    config = load_config(validate=False)
    return HandlerContext(
        config=config,
        operation_id="wrapped.operation",
        _params=params_dict,
    )


def result_from_cli_result(cli_result: CliResult[object]) -> Result[object]:
    """Convert CliResult to Result.

    TEMPORARY: This function will be removed after migration.

    Parameters
    ----------
    cli_result
        CLI result from legacy handler.

    Returns
    -------
    Result[object]
        Operation result.
    """
    if cli_result.success:
        return Result.ok(
            cli_result.data,
            metadata=cli_result.metadata,
            warnings=cli_result.warnings,
        )

    # Convert error
    error = None
    if cli_result.error is not None:
        error = ProblemDetail(
            type=cli_result.error.type,
            title=cli_result.error.title,
            status=cli_result.error.status,
            detail=cli_result.error.detail,
            instance=cli_result.error.instance,
            extensions=cli_result.error.extensions,
        )

    return Result(
        success=False,
        error=error,
        warnings=cli_result.warnings,
        metadata=cli_result.metadata,
    )


def cli_result_from_result(result: Result[object]) -> CliResult[object]:
    """Convert Result to CliResult.

    TEMPORARY: This function will be removed after migration.

    Parameters
    ----------
    result
        Operation result.

    Returns
    -------
    CliResult[object]
        CLI result for legacy code.
    """
    if result.success:
        return CliResult.ok(
            result.data,
            metadata=result.metadata,
        )

    # Convert error
    error = None
    if result.error is not None:
        error = CliProblemDetail(
            type=result.error.type,
            title=result.error.title,
            status=result.error.status,
            detail=result.error.detail,
            instance=result.error.instance,
            extensions=result.error.extensions,
        )

    return CliResult(
        success=False,
        error=error,
        warnings=result.warnings,
        metadata=result.metadata,
    )


__all__ = [
    "HandlerWrapperConfig",
    "cli_result_from_result",
    "operation_from_handler",
    "result_from_cli_result",
]
