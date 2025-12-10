"""Op command operation specifications.

Define and register operations for the op command group including
listing and calling serving operations.
"""

from __future__ import annotations

from codeintel.cli.cli_validation import StringValidator, ValidationSchema
from codeintel.cli.execution import OperationCategory, OperationSpec
from codeintel.cli.handlers.ops import op_list_structured
from codeintel.cli.operation_registry import register_operation
from codeintel.cli.result_types import OperationListResult
from codeintel.cli.results import CliResult


def _op_list_handler(*, category: str | None = None) -> CliResult[OperationListResult]:
    """List available operations handler.

    Parameters
    ----------
    category
        Optional category filter.

    Returns
    -------
    CliResult[OperationListResult]
        List of operations.
    """
    return op_list_structured(category=category)


# Op List Operation
OP_LIST_SPEC: OperationSpec[OperationListResult] = register_operation(
    OperationSpec(
        operation_id="op.list",
        handler=_op_list_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="List available serving operations",
    )
)

# Op Call operation requires runtime context, registered with schema
_op_call_schema = ValidationSchema().add(
    "op_id", StringValidator(min_length=1, pattern=r"^[\w.]+$")
)

__all__ = [
    "OP_LIST_SPEC",
]
