"""Pure compute utilities for validation reporters.

This module provides pure compute functions that extract row data from validation
reporters without performing any I/O. The returned row tuples can be persisted
by the build system using Hamilton materializers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.analytics.parsing.validation import (
        FunctionValidationReporter,
        GraphValidationReporter,
    )


@dataclass(frozen=True)
class ValidationRows:
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
) -> ValidationRows:
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
    ValidationRows
        Container with rows for both validation tables.
    """
    return ValidationRows(
        function_rows=function_reporter.to_rows() if function_reporter else (),
        graph_rows=graph_reporter.to_rows() if graph_reporter else (),
    )


__all__ = [
    "ValidationRows",
    "get_validation_rows",
]
