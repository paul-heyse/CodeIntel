"""Dataset command operation specifications.

Define and register operations for the dataset command group including
list, describe, and verify commands.
"""

from __future__ import annotations

from codeintel.cli.cli_validation import StringValidator, ValidationSchema
from codeintel.cli.execution import OperationCategory, OperationSpec
from codeintel.cli.operation_registry import register_operation
from codeintel.cli.ops_handlers import dataset_describe_structured
from codeintel.cli.result_types import DatasetDescribeResult, DatasetListResult
from codeintel.cli.results import CliResult


def _dataset_list_handler() -> CliResult[DatasetListResult]:
    """List datasets handler.

    Returns
    -------
    CliResult[DatasetListResult]
        List of datasets.

    Notes
    -----
    This is a placeholder handler. The actual implementation requires
    runtime context which is passed from the cyclopts command layer.
    """
    return CliResult.ok(DatasetListResult(datasets=[], count=0))


def _dataset_describe_handler(*, table_key: str) -> CliResult[DatasetDescribeResult]:
    """Describe a dataset handler.

    Parameters
    ----------
    table_key
        Dataset table key.

    Returns
    -------
    CliResult[DatasetDescribeResult]
        Dataset description.
    """
    return dataset_describe_structured(table_key=table_key)


# Dataset List Operation
DATASET_LIST_SPEC: OperationSpec[DatasetListResult] = register_operation(
    OperationSpec(
        operation_id="dataset.list",
        handler=_dataset_list_handler,
        category=OperationCategory.READ,
        param_schema=None,
        requires_progress=False,
        description="List available datasets",
    )
)

# Dataset Describe Operation
_dataset_describe_schema = ValidationSchema().add("table_key", StringValidator(min_length=1))

DATASET_DESCRIBE_SPEC: OperationSpec[DatasetDescribeResult] = register_operation(
    OperationSpec(
        operation_id="dataset.describe",
        handler=_dataset_describe_handler,
        category=OperationCategory.READ,
        param_schema=_dataset_describe_schema,
        requires_progress=False,
        description="Show details for a dataset",
    )
)

__all__ = [
    "DATASET_DESCRIBE_SPEC",
    "DATASET_LIST_SPEC",
]
