"""Pure compute functions and materialization helpers for validation reporters.

This module provides pure compute functions that extract row data from validation
reporters and materialization helpers that integrate with the Hamilton build layer.

Use these functions instead of the deprecated `flush()` methods on the reporter
classes.

Example
-------
>>> from codeintel.analytics.parsing import (
...     FunctionValidationReporter,
...     materialize_function_validation,
... )
>>> reporter = FunctionValidationReporter(repo="org/repo", commit="abc123")
>>> # ... accumulate validation findings ...
>>> ref = materialize_function_validation(ctx, reporter)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.parsing.validation import FUNCTION_VALIDATION_COLS, GRAPH_VALIDATION_COLS
from codeintel.build.hamilton.native.materializer import materialize_rows

if TYPE_CHECKING:
    from codeintel.analytics.parsing.validation import (
        FunctionValidationReporter,
        GraphValidationReporter,
    )
    from codeintel.build.hamilton.native.materializer import DatasetRef, MaterializationContext


@dataclass(frozen=True)
class ValidationResult:
    """Result container for validation reporters.

    Contains row data for both function and graph validation tables without
    performing writes. The rows are tuples matching the column specifications
    in the schema.

    Attributes
    ----------
    function_rows
        Rows for analytics.function_validation table.
    graph_rows
        Rows for analytics.graph_validation table.
    """

    function_rows: tuple[tuple[object, ...], ...]
    graph_rows: tuple[tuple[object, ...], ...]


def get_validation_rows(
    function_reporter: FunctionValidationReporter | None,
    graph_reporter: GraphValidationReporter | None,
) -> ValidationResult:
    """Extract rows from validation reporters without writing.

    Use this function to get accumulated validation rows from reporters
    for materialization via the Hamilton build layer.

    Parameters
    ----------
    function_reporter
        Optional function validation reporter with accumulated rows.
    graph_reporter
        Optional graph validation reporter with accumulated rows.

    Returns
    -------
    ValidationResult
        Container with rows for both validation tables.
    """
    return ValidationResult(
        function_rows=function_reporter.to_rows() if function_reporter else (),
        graph_rows=graph_reporter.to_rows() if graph_reporter else (),
    )


def materialize_function_validation(
    ctx: MaterializationContext,
    reporter: FunctionValidationReporter,
) -> DatasetRef:
    """Materialize function validation rows via Hamilton build layer.

    Use this function instead of `FunctionValidationReporter.flush()` to
    persist validation findings with proper asset catalog tracking.

    Parameters
    ----------
    ctx
        Materialization context with gateway and snapshot info.
    reporter
        Function validation reporter with accumulated rows.

    Returns
    -------
    DatasetRef
        Reference to the materialized dataset with row count.
    """
    return materialize_rows(
        ctx,
        "analytics.function_validation",
        reporter.to_rows(),
        FUNCTION_VALIDATION_COLS,
    )


def materialize_graph_validation(
    ctx: MaterializationContext,
    reporter: GraphValidationReporter,
) -> DatasetRef:
    """Materialize graph validation rows via Hamilton build layer.

    Use this function instead of `GraphValidationReporter.flush()` to
    persist validation findings with proper asset catalog tracking.

    Parameters
    ----------
    ctx
        Materialization context with gateway and snapshot info.
    reporter
        Graph validation reporter with accumulated rows.

    Returns
    -------
    DatasetRef
        Reference to the materialized dataset with row count.
    """
    return materialize_rows(
        ctx,
        "analytics.graph_validation",
        reporter.to_rows(),
        GRAPH_VALIDATION_COLS,
    )


__all__ = [
    "ValidationResult",
    "get_validation_rows",
    "materialize_function_validation",
    "materialize_graph_validation",
]
